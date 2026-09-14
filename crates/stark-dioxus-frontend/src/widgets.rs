//! Small reusable controls shared by the panels, the dialogs and the brush editor.

use std::borrow::Cow;

use crate::commands;
use dioxus::html::{FileData, HasFileData, Key};
use dioxus::prelude::*;
use stark_model::AssetId;
use stark_ui::icons::Icon;

use crate::icons::{icon, label as label_span};
use crate::layout::chrome_dimmed;
use crate::platform;
use crate::preview::Preview;
use crate::state::{AppState, use_obs_opt};
use stark_ui::commands::Command;

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
            {icon(command.icon())}
            {label_span(command.word())}
        }
    }
}

/// The filled share of a range control, as the inline `--fill` custom property
/// the track's gradient is drawn from (`.slider` in stark.css).
///
/// Inline because it is the one part of the slider's look only the control
/// knows: a browser paints the fill only for a *native* range, in the platform's
/// accent blue, and CSS alone cannot see the value. [`SliderTrack`] is the one
/// caller, and `tests/no_raw_slider.rs` keeps it so.
fn slider_fill(min: f32, max: f32, value: f32) -> String {
    let pct = if max > min {
        ((value - min) / (max - min) * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    format!("--fill: {pct}%")
}

/// A control's mark, then its word — the word wrapped as hideable
/// ([`crate::icons::label`]) only when there is a mark to survive it. A control with
/// neither would be anonymous in minimal mode, so an unmarked one keeps its word; one
/// function, so a slider, a chip and a bar cannot be given the wrong pair.
fn mark_and_word(glyph: Option<Icon>, word: &str) -> Element {
    match glyph {
        Some(glyph) => rsx! { {icon(glyph)} {label_span(word)} },
        None => rsx! { "{word}" },
    }
}

/// What a track stands in, which decides the markup around it — each is keyed by its
/// own classes in the stylesheet, so a site picks one rather than writing the shell.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum SliderShape {
    /// A row of a panel or dialog: the mark and word over the track (`.slider-row`).
    #[default]
    Row,
    /// Inline in a bottom bar: the mark and word a `.bar-sub` beside the track.
    Bar,
    /// A line of the filter bar's knob grid (`.filter-knob`): the word and the readout
    /// side by side over the track.
    Knob,
}

/// A labelled range control.
///
/// `glyph` is an `Option` because the brush editor's parameter list is not marked yet,
/// not because a slider may go without one: a `None` is a to-do (see
/// [`stark_ui::icons::SIZE`]). `marked` carries the same fact to the stylesheet, which
/// folds a marked row onto one line in minimal mode — an unmarked row keeps its word and
/// cannot fold, or the tracks would start at a ragged edge.
///
/// `readout` is the number beside the word, never inside it: minimal mode takes a
/// control's name, not its value. `title` goes on the whole line where there is one,
/// and on the track in a bar, where the mark and word are a sibling rather than a row.
///
/// `disabled` greys the track out for a value with nothing to act on, as a chip is.
/// The rest is [`SliderTrack`]'s.
#[component]
pub fn Slider(
    label: String,
    #[props(default)] glyph: Option<Icon>,
    min: f32,
    max: f32,
    value: f32,
    /// What the track snaps to, or `None` for continuous.
    #[props(default)]
    step: Option<f32>,
    #[props(default)] readout: Option<String>,
    #[props(default)] title: Option<String>,
    #[props(default)] disabled: bool,
    #[props(default)] shape: SliderShape,
    oninput: EventHandler<f32>,
    #[props(default)] onsettle: Option<EventHandler<()>>,
) -> Element {
    let name = mark_and_word(glyph, &label);
    let (line_title, track_title) = match shape {
        SliderShape::Bar => (None, title),
        SliderShape::Row | SliderShape::Knob => (title, None),
    };
    let track = rsx! {
        SliderTrack { min, max, value, step, title: track_title, disabled, oninput, onsettle }
    };
    match shape {
        SliderShape::Row => rsx! {
            div {
                class: if glyph.is_some() { "slider-row marked" } else { "slider-row" },
                title: line_title,
                div { class: "slider-label", {name} {readout} }
                {track}
            }
        },
        SliderShape::Bar => rsx! {
            span { class: "bar-sub", {name} {readout} }
            {track}
        },
        SliderShape::Knob => rsx! {
            div { class: "filter-knob", title: line_title,
                span { class: "filter-knob-label", {name} }
                span { class: "filter-knob-value", {readout} }
                {track}
            }
        },
    }
}

/// The range input every slider is: its fill, its step, and the drag's end.
///
/// `onsettle` is wired to all three events that can end a drag
/// ([`Preview::settle`] says why one is not enough). A slider whose value is document
/// state is a [`PreviewSlider`], which cannot preview without settling.
///
/// Drawn bare only by a row whose name is not a [`Slider`]'s — a settings row, whose
/// `<label for>` points at `id` and whose sentence stands between the name and the
/// track. `class` sits beside `slider`.
#[component]
pub fn SliderTrack(
    min: f32,
    max: f32,
    value: f32,
    #[props(default)] step: Option<f32>,
    #[props(default)] id: Option<String>,
    #[props(default)] class: &'static str,
    #[props(default)] title: Option<String>,
    #[props(default)] disabled: bool,
    oninput: EventHandler<f32>,
    #[props(default)] onsettle: Option<EventHandler<()>>,
) -> Element {
    let settle = move || {
        if let Some(h) = &onsettle {
            h.call(());
        }
    };
    let step = step.map_or_else(|| "any".to_string(), |s| s.to_string());
    rsx! {
        input {
            id,
            class: "slider {class}",
            style: slider_fill(min, max, value),
            r#type: "range", min: "{min}", max: "{max}", step, value: "{value}",
            title,
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

/// A [`Slider`] over **document state**: every sample is shown through `preview` and
/// the value is laid down once, when the drag settles (`crate::preview`) — one undo step
/// per adjustment rather than one per pointer move.
///
/// `map` turns a track position into the value, or `None` when there is nothing left to
/// show it on (a guide removed under the hand). `pending` is the site's, since one bar's
/// knobs and pictures can share it. The three settle events are this component's, so a
/// site that previews cannot forget to lay down.
#[component]
pub fn PreviewSlider<T: Clone + 'static>(
    preview: Preview<T>,
    pending: Signal<Option<T>>,
    map: Callback<f32, Option<T>>,
    label: String,
    #[props(default)] glyph: Option<Icon>,
    min: f32,
    max: f32,
    value: f32,
    #[props(default)] step: Option<f32>,
    #[props(default)] readout: Option<String>,
    #[props(default)] title: Option<String>,
    #[props(default)] disabled: bool,
    #[props(default)] shape: SliderShape,
) -> Element {
    let state = use_context::<AppState>();
    rsx! {
        Slider {
            label, glyph, min, max, value, step, readout, title, disabled, shape,
            oninput: move |v| {
                if let Some(value) = map.call(v) {
                    preview.during(state, pending, value);
                }
            },
            onsettle: move |()| preview.settle(state, pending),
        }
    }
}

/// The chrome's button, lit while what it names is in force.
///
/// `class` is a chip class of its own beside `chip`, for the few drawn differently
/// (`axis-chip`, `gradient-trace`). `title` is not optional: a chip is mostly a mark,
/// and the tooltip is where it says what it does.
#[component]
pub fn Chip(
    #[props(default)] active: bool,
    #[props(default)] disabled: bool,
    title: String,
    #[props(default)] class: &'static str,
    #[props(default)] style: Option<String>,
    onclick: EventHandler<MouseEvent>,
    children: Element,
) -> Element {
    rsx! {
        button {
            class: "chip",
            class: "{class}",
            class: if active { "active" },
            disabled,
            title,
            style,
            onclick: move |e| onclick.call(e),
            {children}
        }
    }
}

/// What a chip in a [`Segmented`] run shows.
#[derive(Clone, PartialEq)]
pub enum Face {
    /// A mark and its word, the word hideable in minimal mode.
    Marked(Icon, &'static str),
    /// A word alone, which minimal mode leaves standing — a speed, a patch size.
    Word(Cow<'static, str>),
    /// Drawn by the site, for the face the catalog cannot give: a bucket full of the
    /// paint it would lay.
    Drawn(Element),
}

impl Face {
    fn draw(self) -> Element {
        match self {
            Face::Marked(glyph, word) => mark_and_word(Some(glyph), word),
            Face::Word(word) => mark_and_word(None, &word),
            Face::Drawn(drawn) => drawn,
        }
    }
}

/// One answer in a [`Segmented`] run.
#[derive(Clone, PartialEq)]
pub struct Choice<V> {
    pub value: V,
    pub face: Face,
    pub tip: String,
    /// [`Chip`]'s own class.
    pub class: &'static str,
    pub style: Option<String>,
}

impl<V> Choice<V> {
    /// `value` wearing `face`, explained by `tip`, with no class or style of its own —
    /// the rest is struct update at the site.
    pub fn new(value: V, face: Face, tip: impl Into<String>) -> Self {
        Self {
            value,
            face,
            tip: tip.into(),
            class: "",
            style: None,
        }
    }
}

/// A run of chips answering one question, which is one control (§25.9): it always wears
/// `.segmented`, whose closed seams say that picking one un-picks the rest. `class` is
/// the run's classes beside that.
///
/// The lit chip is the one whose value is `selected` — one answer for the run, so a run
/// cannot light two. Switches that may be held together are separate [`Chip`]s.
///
/// A pick hands its value to `onpick`; what re-picking the lit chip means is the site's.
#[component]
pub fn Segmented<V: Clone + PartialEq + 'static>(
    choices: Vec<Choice<V>>,
    selected: V,
    onpick: EventHandler<V>,
    #[props(default)] class: &'static str,
) -> Element {
    rsx! {
        div { class: "segmented {class}",
            // Keyed by place: a run is a fixed set of answers and never reorders.
            for (i, Choice { value, face, tip, class: own, style }) in choices.into_iter().enumerate() {
                Chip {
                    key: "{i}",
                    active: value == selected,
                    title: tip,
                    class: own,
                    style,
                    onclick: move |_| onpick.call(value.clone()),
                    {face.draw()}
                }
            }
        }
    }
}

/// A drop-down over a list of names, answering with the index of the one picked — the
/// ladder's rung for a choice too wide for a [`Segmented`] run (§25.9).
///
/// Each option's value is its index, so two options with one name still answer apart.
#[component]
pub fn Select(
    options: Vec<&'static str>,
    selected: Option<usize>,
    #[props(default)] title: Option<String>,
    #[props(default)] disabled: bool,
    onchange: EventHandler<usize>,
) -> Element {
    let count = options.len();
    rsx! {
        select {
            class: "select",
            title,
            disabled,
            onchange: move |e| {
                if let Ok(i) = e.value().parse::<usize>()
                    && i < count
                {
                    onchange.call(i);
                }
            },
            for (i, name) in options.into_iter().enumerate() {
                option { value: "{i}", selected: selected == Some(i), "{name}" }
            }
        }
    }
}

/// A bottom bar (MODAL_DESIGN.md): its mark and name, then its controls.
///
/// `mode` marks the composing mode's own bar (`.mode-bar`). Every other bar **recedes**
/// while a mode composes — dimmed and inert (`.recessed` takes the pointer), but on
/// screen, so the place the mode's Done and Esc return to stays in view. A mode's own
/// bar never does; the gradient bar parked behind a trace is not a mode's bar while it
/// is parked.
#[component]
pub fn Bar(
    class: &'static str,
    glyph: Icon,
    word: String,
    #[props(default)] mode: bool,
    children: Element,
) -> Element {
    let state = use_context::<AppState>();
    let recessed = !mode && crate::modes::composing(state).is_some();
    rsx! {
        div {
            class: "{class}",
            class: if mode { "mode-bar" },
            class: "chrome",
            class: if chrome_dimmed(state) { "dimmed" },
            class: if recessed { "recessed" },
            span { class: "bar-label", {mark_and_word(Some(glyph), &word)} }
            {children}
        }
    }
}

/// A bar's **Done**: a chip that wears a registry command's chord but runs the bar's own
/// act — not a [`CommandButton`], because the command is how the *keyboard* reaches that
/// act, while the act the chip runs belongs to the bar drawing it.
///
/// Worded as `Command::FinishMode` is, whichever chord it advertises: a filter's or a
/// frame's Done is Esc's act, but every edit on those bars is laid as it is made, so
/// leaving keeps them and the chip is still a Done.
#[component]
pub fn ActChip(command: Command, title: String, onclick: EventHandler<MouseEvent>) -> Element {
    let state = use_context::<AppState>();
    let finish = Command::FinishMode;
    rsx! {
        Chip {
            title: stark_ui::commands::advertised(&title, command, &state.bindings.read()),
            onclick,
            {icon(finish.icon())}
            {label_span(finish.word())}
        }
    }
}

/// The name field a double-click opens over a roster row — a layer's, a guide's, a
/// gradient's.
///
/// It opens on `seed`, the row's *name*: seeding with the description standing in for a
/// missing one would turn opening the field and pressing Enter into naming the row
/// "Layer 3". `placeholder` says what the row is called meanwhile. It takes the keyboard
/// with its text selected, since the usual reason to open it is to replace the name.
///
/// **Blur** and **Enter** commit — Enter directly, since a focused element that is
/// removed does not reliably fire `blur` — and **Escape** abandons. Whichever runs first
/// *takes* the draft, so the blur that follows a field closed by a key finds nothing.
/// What else is typed is the field's: the global shortcuts stand aside for a text field
/// (`input::bind_shortcuts`), and a click in it places the caret rather than reaching the
/// row beneath.
#[component]
pub fn InlineRename(
    class: &'static str,
    seed: String,
    #[props(default)] placeholder: Option<String>,
    oncommit: EventHandler<String>,
    onclose: EventHandler<()>,
) -> Element {
    let mut draft = use_signal(move || Some(seed));
    let mut close = move |keep: bool| {
        let Some(text) = draft.write().take() else {
            return;
        };
        if keep {
            oncommit.call(text);
        }
        onclose.call(());
    };
    let text = draft().unwrap_or_default();
    rsx! {
        input {
            class,
            r#type: "text",
            value: "{text}",
            placeholder,
            onmounted: move |e| focus_selected(&e),
            oninput: move |e| draft.set(Some(e.value())),
            onclick: move |e| e.stop_propagation(),
            ondoubleclick: move |e| e.stop_propagation(),
            onblur: move |_| close(true),
            onkeydown: move |e| match e.key() {
                Key::Enter => close(true),
                Key::Escape => close(false),
                _ => {}
            },
        }
    }
}

/// What a gallery card shows where its picture goes.
#[derive(Clone, PartialEq, Debug)]
pub enum Thumb {
    /// A `data:` URL, or `None` while there is none yet — a built-in still fetching.
    Picture(Option<String>),
    /// No picture, and the flat ground drawn in its place (`.asset-thumb.flat`).
    Flat,
    /// The procedural round tip, which the stylesheet draws (`.asset-thumb.round`).
    Round,
}

impl Thumb {
    fn class(&self) -> &'static str {
        match self {
            Thumb::Picture(_) => "asset-thumb",
            Thumb::Flat => "asset-thumb flat",
            Thumb::Round => "asset-thumb round",
        }
    }

    /// The inline picture. `None` only for the round tip, whose disc is a `background`
    /// an inline `background-image` would override.
    fn style(&self) -> Option<String> {
        match self {
            Thumb::Picture(url) => Some(crate::cards::thumb_style(url.as_deref())),
            Thumb::Flat => Some(crate::cards::thumb_style(None)),
            Thumb::Round => None,
        }
    }
}

/// One card in an [`AssetGallery`].
#[derive(Clone, PartialEq, Debug)]
pub struct AssetCard<V> {
    /// Unique among the gallery's cards, and stable for the asset it shows.
    pub key: String,
    pub name: String,
    pub thumb: Thumb,
    /// Wears the selected ring.
    pub selected: bool,
    /// The card's hover, where it has one; a card without one puts its name there, which
    /// the grid may have cut short.
    pub blurb: Option<&'static str>,
    /// What a press on the card picks, or `None` for a card that cannot be picked.
    pub pick: Option<V>,
    /// The library id a remove button takes out, on a card the user's library holds.
    pub remove: Option<AssetId>,
}

/// A grid of asset cards — the brush editor's stamps, the Lighting panel's surfaces —
/// with an import card at its end, the library's notice under it and an optional hint.
///
/// Every handler is called synchronously from the event it answers, so nothing here
/// spawns and nothing outlives the gallery: `onimport` runs inside the click gesture a
/// file picker needs, and the imports a caller starts are its own `spawn_forever`s.
///
/// A drop on the grid is claimed (`stopPropagation`): the app root places every other
/// dropped file as a picture (§23.4), and adding one to a library is a different act.
#[component]
pub fn AssetGallery<V: Clone + PartialEq + 'static>(
    cards: Vec<AssetCard<V>>,
    notice: Option<String>,
    #[props(default)] hint: Option<&'static str>,
    onpick: EventHandler<V>,
    onremove: EventHandler<AssetId>,
    onimport: EventHandler<()>,
    ondrop: EventHandler<Vec<FileData>>,
) -> Element {
    let mut dropping = use_signal(|| false);
    rsx! {
        div {
            class: if dropping() { "asset-grid dropping" } else { "asset-grid" },
            ondragover: move |e| {
                e.prevent_default();
                e.stop_propagation();
                dropping.set(true);
            },
            ondragleave: move |_| dropping.set(false),
            ondrop: move |e| {
                e.prevent_default();
                e.stop_propagation();
                dropping.set(false);
                ondrop.call(e.files());
            },
            for AssetCard { key, name, thumb, selected, blurb, pick, remove } in cards {
                div {
                    key: "{key}",
                    class: if selected { "asset-card selected" } else { "asset-card" },
                    title: blurb,
                    onclick: move |_| {
                        if let Some(pick) = &pick {
                            onpick.call(pick.clone());
                        }
                    },
                    div { class: thumb.class(), style: thumb.style() }
                    div {
                        class: "asset-name",
                        title: blurb.is_none().then(|| name.clone()),
                        "{name}"
                    }
                    if let Some(id) = remove {
                        button {
                            class: "asset-remove",
                            title: "Remove from library",
                            onclick: move |e| {
                                e.stop_propagation();
                                onremove.call(id);
                            },
                            {icon(stark_ui::icons::REMOVE)}
                        }
                    }
                }
            }
            div { class: "asset-card import",
                onclick: move |_| onimport.call(()),
                div { class: "asset-thumb plus", {icon(stark_ui::icons::ADD)} }
                div { class: "asset-name", "Import\u{2026}" }
            }
        }
        if let Some(notice) = notice {
            div { class: "asset-notice", "{notice}" }
        }
        if let Some(hint) = hint {
            div { class: "asset-hint", "{hint}" }
        }
    }
}

/// Take the keyboard and select everything in the field `e` was mounted on — for a field
/// opened to replace what it holds. Synchronous, so the selection cannot land before the
/// focus does.
pub fn focus_selected(e: &Event<MountedData>) {
    platform::focus(e);
    platform::select_all(e);
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
/// on `stark_ui::modes::Composing`'s argument: two open at once is a state nothing wants
/// and nothing should have to prevent. They were `use_signal(|| false)` locals of
/// the surfaces that draw them, which made them invisible to the app — and in
/// particular to Escape, whose ladder knows the dialogs, the composing modes, the
/// composing layers and Timeline mode, and could not see a pop-out standing over
/// all of them (`commands::escape`). That is why the rail's menu is on this list
/// though no well opens it: reaching it from the ladder is the only way its
/// Escape is not a *second* handler for a keystroke the window already hears.
///
/// **Not on the dialog stack**, though the machinery would
/// have fitted: that list is also what stands `FinishMode` down
/// (`dialogs::any_open`), and the gradient library is opened *from* the
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
#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::VariantArray)]
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
    /// What the row this pop-out flies out of wears as its `data-popout`.
    pub const fn key(self) -> &'static str {
        match self {
            PopoutId::VisibilityMenu => "visibility-menu",
            PopoutId::Parcel => "parcel",
            PopoutId::GradientLibrary => "gradient-library",
            PopoutId::SubstrateColor => "substrate-color",
            PopoutId::SubstrateGallery => "substrate-gallery",
        }
    }

    /// That row as a selector, `[data-popout="<key>"]`. Spelled out, and held to
    /// [`key`](Self::key) by `a_pop_out_is_found_by_its_own_key`.
    const fn selector(self) -> &'static str {
        match self {
            PopoutId::VisibilityMenu => r#"[data-popout="visibility-menu"]"#,
            PopoutId::Parcel => r#"[data-popout="parcel"]"#,
            PopoutId::GradientLibrary => r#"[data-popout="gradient-library"]"#,
            PopoutId::SubstrateColor => r#"[data-popout="substrate-color"]"#,
            PopoutId::SubstrateGallery => r#"[data-popout="substrate-gallery"]"#,
        }
    }

    /// The row this pop-out flies out of, as a selector — and `None` for one that is
    /// drawn in place inside the bar that owns it.
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
            PopoutId::SubstrateColor | PopoutId::SubstrateGallery => Some(self.selector()),
        }
    }
}

/// Whether `id` is the pop-out currently open. Subscribing — the caller is the
/// bar that mounts it.
pub fn popout_open(state: AppState, id: PopoutId) -> bool {
    *state.popout.read() == Some(id)
}

/// Whether `id` is open, for the surface that owns it — and that surface's promise to
/// put it down on the way out (§25.7). A pop-out is drawn by or placed against the
/// surface that opened it, so one that unmounted with the flag standing would be open
/// on whatever came up next. A hook: call it unconditionally, above any early return.
pub fn use_popout(state: AppState, id: PopoutId) -> bool {
    use_drop(move || close_popout_of(state, id));
    popout_open(state, id)
}

/// Open `id`, closing whichever pop-out was open. Toggles, since every one of
/// these is opened by a press on the well it flies out of.
pub fn toggle_popout(state: AppState, id: PopoutId) {
    let mut open = state.popout;
    let was = *open.peek();
    open.set(if was == Some(id) { None } else { Some(id) });
}

/// Close `id` if it is the one open — what a surface that light-dismisses calls
/// when it hears the press leave it (`rail::VisibilityMenu`), and what
/// [`use_popout`] calls on the way out.
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
pub fn close_popout(state: AppState) -> bool {
    let mut open = state.popout;
    let was = open.peek().is_some();
    if was {
        open.set(None);
    }
    was
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use strum::VariantArray;

    /// A stack pop-out is placed against the row wearing its key, so the selector it
    /// is found by has to be that key's — for every id, and the stack's in particular.
    #[test]
    fn a_pop_out_is_found_by_its_own_key() {
        for id in PopoutId::VARIANTS {
            let row = format!("[data-popout=\"{}\"]", id.key());
            assert_eq!(id.selector(), row, "{id:?}");
            if let Some(selector) = id.in_stack() {
                assert_eq!(selector, row, "{id:?}");
            }
        }
    }

    /// Two pop-outs wearing one key would place one against the other's row.
    #[test]
    fn no_two_pop_outs_share_a_key() {
        let keys: HashSet<_> = PopoutId::VARIANTS.iter().map(|id| id.key()).collect();
        assert_eq!(keys.len(), PopoutId::VARIANTS.len());
    }

    /// Every run wears `.segmented` beside its own class, since that class is the run's
    /// whole shape (§25.9). Rendered rather than read: the spelling that lost it compiles
    /// (`tests/no_split_class_strings.rs`).
    #[test]
    fn a_run_wears_segmented_beside_its_own_class() {
        fn app() -> Element {
            rsx! {
                Segmented {
                    class: "setting-choice",
                    choices: vec![Choice::new(1u8, Face::Drawn(rsx! { "a" }), "")],
                    selected: 1u8,
                    onpick: |_| {},
                }
                Segmented {
                    choices: vec![Choice::new(1u8, Face::Drawn(rsx! { "b" }), "")],
                    selected: 1u8,
                    onpick: |_| {},
                }
            }
        }
        let runs: Vec<_> = rendered_classes(app)
            .into_iter()
            .filter(|classes| !classes.iter().any(|c| c == "chip"))
            .collect();
        assert_eq!(
            runs,
            [vec!["segmented", "setting-choice"], vec!["segmented"]]
        );
    }

    /// Every track wears `.slider`, which is its width, its neutral paint and its thumb;
    /// without it the browser draws its own blue range input.
    #[test]
    fn a_track_wears_slider_beside_its_own_class() {
        fn app() -> Element {
            rsx! {
                SliderTrack { min: 0.0, max: 1.0, value: 0.5, class: "setting-slider", oninput: |_| {} }
                SliderTrack { min: 0.0, max: 1.0, value: 0.5, oninput: |_| {} }
            }
        }
        assert_eq!(
            rendered_classes(app),
            [vec!["slider", "setting-slider"], vec!["slider"]]
        );
    }

    /// A gallery's grid, its cards and their pictures wear the classes the stylesheet
    /// draws them by: the selected ring, and the flat ground and round tip that stand in
    /// for a picture.
    #[test]
    fn a_gallery_card_wears_its_ring_and_its_picture() {
        fn app() -> Element {
            let card = |key: &str, thumb: Thumb, selected: bool| AssetCard {
                key: key.to_string(),
                name: key.to_string(),
                thumb,
                selected,
                blurb: None,
                pick: Some(0u8),
                remove: None,
            };
            rsx! {
                AssetGallery {
                    cards: vec![
                        card("picture", Thumb::Picture(None), true),
                        card("flat", Thumb::Flat, false),
                        card("round", Thumb::Round, false),
                    ],
                    notice: None,
                    onpick: |_| {},
                    onremove: |_| {},
                    onimport: |_| {},
                    ondrop: |_| {},
                }
            }
        }
        assert_eq!(
            rendered_classes(app),
            [
                vec!["asset-grid"],
                vec!["asset-card", "selected"],
                vec!["asset-thumb"],
                vec!["asset-card"],
                vec!["asset-thumb", "flat"],
                vec!["asset-card"],
                vec!["asset-thumb", "round"],
            ]
        );
    }

    /// The `class` each element `app` renders is given, in render order, split into names.
    fn rendered_classes(app: fn() -> Element) -> Vec<Vec<String>> {
        VirtualDom::new(app)
            .rebuild_to_vec()
            .edits
            .into_iter()
            .filter_map(|edit| match edit {
                dioxus::core::Mutation::SetAttribute {
                    name: "class",
                    value: dioxus::core::AttributeValue::Text(text),
                    ..
                } => Some(text.split_whitespace().map(str::to_owned).collect()),
                _ => None,
            })
            .collect()
    }
}
