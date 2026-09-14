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
/// tooltip (`crate::commands`), so a control and its act cannot describe each other
/// differently.
///
/// `class` is the call site's, since the registry does not say where an act is drawn;
/// what the button says or does is not. A site that must vary that (the Fill chip's
/// tinted bucket) writes its own `button` and still reads the words off the command.
///
/// Lit and greyed come off [`Command::active`] and [`Command::enabled`], not props, so a
/// chip agrees with the menu and palette. A bar can stand before what its acts need
/// exists (§6.8); `run` still gates the act.
#[component]
pub fn CommandButton(
    command: Command,
    #[props(default = String::from("chip"))] class: String,
) -> Element {
    let state = use_context::<AppState>();
    // Memoized: the projection moves at pointer rate during a stroke, and these two bools
    // almost never do.
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
/// `.slider` draws its track from. Inline because CSS cannot see a range's value;
/// [`SliderTrack`] is the one caller (`tests/no_raw_slider.rs`).
fn slider_fill(min: f32, max: f32, value: f32) -> String {
    let pct = if max > min {
        ((value - min) / (max - min) * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };
    format!("--fill: {pct}%")
}

/// A control's mark, then its word — the word hideable ([`crate::icons::label`]) only
/// when there is a mark to survive it, so no control is anonymous in minimal mode.
fn mark_and_word(glyph: Option<Icon>, word: &str) -> Element {
    match glyph {
        Some(glyph) => rsx! { {icon(glyph)} {label_span(word)} },
        None => rsx! { "{word}" },
    }
}

/// What a track stands in, which decides the markup around it.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum SliderShape {
    /// A row of a panel or dialog: the mark and word over the track (`.slider-row`).
    #[default]
    Row,
    /// Inline in a bottom bar: the mark and word a `.bar-sub` beside the track.
    Bar,
    /// A line of the filter bar's knob grid (`.filter-knob`): word and readout over the track.
    Knob,
}

/// A labelled range control.
///
/// A `None` `glyph` is a to-do (see [`stark_ui::icons::SIZE`]). A marked row folds onto
/// one line in minimal mode; an unmarked one keeps its word and cannot, or the tracks
/// would start at a ragged edge. `readout` sits beside the word, never inside it, since
/// minimal mode hides the name and not the value. `title` goes on the whole line, or on
/// the track in a [`SliderShape::Bar`].
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
/// `onsettle` fires on all three events that can end a drag ([`Preview::settle`] says
/// why). Drawn bare only by a settings row, whose `<label for>` points at `id`.
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

/// A [`Slider`] over **document state**: every sample is shown through `preview` and laid
/// down once, when the drag settles (`crate::preview`) — one undo step per adjustment.
///
/// `map` answers `None` when there is nothing left to show the value on (a guide removed
/// under the hand). `pending` is the site's, since one bar's knobs can share it.
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
/// `class` sits beside `chip`. `title` is required: a chip is mostly a mark, and the
/// tooltip is where it says what it does.
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
    /// Drawn by the site, for a face the catalog cannot give (a bucket full of its paint).
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
    /// `value` wearing `face`, explained by `tip`, with no class or style of its own.
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

/// A run of chips answering one question (§25.9), worn as `.segmented` beside `class`.
///
/// The lit chip is the one whose value is `selected`, so a run cannot light two;
/// switches that may be held together are separate [`Chip`]s. What re-picking the lit
/// chip means is the site's.
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

/// A drop-down answering with the index picked, for a choice too wide for a
/// [`Segmented`] run (§25.9). Indices, so two options with one name still answer apart.
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
/// While a mode composes, every bar but the mode's own (`mode`) recedes: dimmed and inert,
/// but on screen, so the place Done and Esc return to stays in view. A bar parked behind
/// a trace is not the mode's bar.
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

/// A bar's **Done**: wears `command`'s chord but runs the bar's own act, so it is not a
/// [`CommandButton`]. Always worded as `Command::FinishMode`: even where the chord is
/// Esc's, the bar's edits are laid as they are made, so leaving keeps them.
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

/// The name field a double-click opens over a roster row.
///
/// Seeded with the row's *name*, never its description, or Enter on an untouched field
/// would name the row "Layer 3". Opens focused with its text selected.
///
/// **Blur** and **Enter** commit — Enter directly, since a focused element that is
/// removed does not reliably fire `blur` — and **Escape** abandons. Whichever runs first
/// *takes* the draft, so the blur after a key-close finds nothing. A click or double-click
/// stops here, so it places the caret rather than reaching the row beneath.
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

    /// The inline picture, and nothing without one: the flat ground and the round tip are
    /// stylesheet `background`s that an inline `background-image` would override.
    fn style(&self) -> Option<String> {
        match self {
            Thumb::Picture(Some(url)) => Some(crate::cards::thumb_style(Some(url))),
            Thumb::Picture(None) | Thumb::Flat | Thumb::Round => None,
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
    /// The card's hover; without one the name goes there, since the grid may cut it short.
    pub blurb: Option<&'static str>,
    /// What a press on the card picks, or `None` for a card that cannot be picked.
    pub pick: Option<V>,
    /// The library id a remove button takes out, on a card the user's library holds.
    pub remove: Option<AssetId>,
}

/// A grid of asset cards with an import card at its end, the library's notice under it
/// and an optional hint.
///
/// No `spawn` inside: every handler runs synchronously in its event, so `onimport` is
/// inside the click gesture a file picker needs; an import the caller starts is its own
/// `spawn_forever`. A drop on the grid stops propagating,
/// since the app root places any other dropped file as a picture (§23.4).
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

/// Focus the field `e` was mounted on and select its contents. Synchronous, so the
/// selection cannot land before the focus.
pub fn focus_selected(e: &Event<MountedData>) {
    platform::focus(e);
    platform::select_all(e);
}

/// The shell every dialog floats in: the dimmed backdrop, the box on it, and the
/// press-outside-to-dismiss rule (§25.7).
///
/// `class` and the spread attributes land on the box. `on_close` is `None` only for the
/// GPU-failure notice ([`crate::failure`]), which has nothing behind it to go back to.
///
/// **A click dismisses only if the backdrop also heard its `pointerdown`.** A menu row
/// opens a dialog on `pointerdown`, and for a pen or touch the browser delivers that
/// press's `mousedown`, `mouseup` and `click` at the release point — onto the backdrop
/// the press just created. A mouse fires no click once its press target is gone, so only a
/// pen or touch shows this. The box stops `pointerdown` and `click`, so a drag out of the
/// dialog released over the dim does not dismiss either; the window-wide press listener
/// binds in the capture phase and still hears them ([`crate::platform::on_window_pointer`]).
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
            // Terminal: a stale arm would be spent on whatever click came next.
            onpointercancel: move |_| armed.set(false),
            onclick: move |_| {
                // Spent on every click, whether or not it dismisses.
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

/// The pop-outs the chrome can fly open: surfaces that are neither a panel nor a dialog
/// (§25.7).
///
/// One at a time, in one signal ([`AppState::popout`](crate::state::AppState)), so
/// Escape's ladder can see them (`commands::escape`). Not on the dialog stack: that stack
/// stands `FinishMode` down (`dialogs::any_open`), and the gradient library opens while a
/// fill composes, so Enter's Done must survive it.
///
/// # What is still owed
///
/// **Light dismiss** for pop-outs flown out of a bar or a panel; only the rail's menu has it
/// (`focusout`, `rail::VisibilityMenu`). A catcher must be root-mounted like [`Modal`]'s
/// backdrop, since `.bottom-bars`' `transform` and each bar's `backdrop-filter` trap a
/// `position: fixed` child; check its z-index in a browser. A stack pop-out already closes
/// on a canvas gesture (`panels::popout`, `StackPopouts`), the press that matters there.
#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::VariantArray)]
pub enum PopoutId {
    /// The rail's map of what is on screen (§25.5). Not a well's: here for Escape's rung
    /// alone, so the menu is not a second handler for a key the window already hears
    /// (`rail::VisibilityMenu`, `input::keys`).
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

    /// The row this pop-out flies out of, as a selector, or `None` for one drawn in place
    /// inside the bar that owns it.
    ///
    /// Nothing clips `.bottom-bars`, so a bar hangs its pop-out off the well. The panel stack
    /// clips and every panel has a `backdrop-filter`, so a panel's pop-out is mounted at the
    /// app root and placed against this row (`panels::popout::StackPopouts`,
    /// `crate::anchor`). The row rather than the well, so its left edge is the panel's own.
    pub fn in_stack(self) -> Option<&'static str> {
        match self {
            PopoutId::VisibilityMenu | PopoutId::Parcel | PopoutId::GradientLibrary => None,
            PopoutId::SubstrateColor | PopoutId::SubstrateGallery => Some(self.selector()),
        }
    }
}

/// Whether `id` is the pop-out currently open. Subscribes.
pub fn popout_open(state: AppState, id: PopoutId) -> bool {
    *state.popout.read() == Some(id)
}

/// Whether `id` is open, closing it when the owning surface unmounts (§25.7) — otherwise
/// it would stay open on whatever came up next. A hook: call it above any early return.
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

/// Close `id` if it is the one open: on a light dismiss (`rail::VisibilityMenu`) and in
/// [`use_popout`].
///
/// Guarded, unlike [`close_popout`], because the dismissing press often opens the next
/// pop-out, and focus leaves before the new well's click lands. `peek` rather than `read`,
/// so a handler is not subscribed to the signal it writes.
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

    /// Every run wears `.segmented` beside its own class (§25.9). Rendered rather than read:
    /// on Dioxus 0.7.10, `class: "a", class: "{b}"` compiles and renders an EMPTY class
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

    /// A gallery's grid, cards and pictures wear the classes the stylesheet draws them by.
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
