//! The command rail down the far left (§11): the search palette, the visibility
//! menu, the ⚙, and the dialogs they open.
//!
//! Everything here **renders a command rather than restating one**
//! (`crate::commands`): a row's word, mark, shortcut column, greyed state and
//! tick are all the registry's, so the three ways to the same act — a menu row,
//! a palette row, a chord — cannot describe it differently or forget its gate.
//! That is §25.2's rule, and this is the module with the most rows to keep it
//! for.
//!
//! The palette is our own dropdown rather than a third `MenubarMenu`, and not
//! for styling: the primitive light-dismisses the moment DOM focus leaves it for
//! anything but a menu item, and the whole point of this surface is that focus
//! lives in a text field the primitive has never heard of. [`CommandSearch`]
//! carries the arrangement that replaces it.

use dioxus::html::{Key, Modifiers};
use dioxus::prelude::*;

use crate::commands::{self, Command, VisibilityToggle};
use crate::components::menubar::{
    Menubar, MenubarContent, MenubarItem, MenubarMenu, MenubarTrigger,
};
use crate::credits::CreditsModal;
use crate::icons::{self, icon, icon_large};
use crate::input::accel;
use crate::layout::chrome_class;
use crate::platform;
use crate::settings::SettingsModal;
use crate::state::{AppState, use_obs_opt};
use crate::substrates::NewDocumentModal;
use crate::{collab, drags, files, timings};

/// A vertical rail on the far left (§11): the command search, the visibility
/// menu — what is on screen, panel or not (`commands::VisibilityToggle`) — and
/// the ⚙. The menu is the `menubar` component and its dropdown flies out to
/// the right; the search is [`CommandSearch`], our own dropdown in the same
/// spot, which is the way to every simple command by name — Undo advertises its
/// Ctrl+Z there now, in the row a query for it turns up.
///
/// The rail ends in a ⚙ that opens [`SettingsModal`] directly rather than dropping
/// a menu: settings are a *destination*, not a list of commands to pick one from,
/// and the one thing that menu would ever contain is the dialog itself.
#[component]
pub fn CommandRail() -> Element {
    let state = use_context::<AppState>();
    // The dialogs' flags are app state (`state::Dialogs`), raised by the
    // commands that open them — which is what lets the same act be a menu row
    // today and whatever reaches for it tomorrow. Local names for the mounts
    // and their `on_close` below; nothing in this component sets one `true`.
    let mut show_new_doc = state.dialogs.new_document;
    let mut show_session = state.dialogs.session;
    let mut show_export = state.dialogs.export;
    let mut show_settings = state.dialogs.settings;
    let mut show_timing = state.dialogs.timing;
    let mut show_credits = state.dialogs.credits;
    // The one flag here nothing in the chrome raises: the drag-preset offer is
    // put up by a canvas release (`drags::settle_offer`, §25.8). Mounted beside
    // the rest all the same, because where a root dialog lives is
    // `AppState::root_dialogs`' question — Esc lowers this one like any other.
    let mut show_drag_presets = state.dialogs.drag_presets;

    rsx! {
        div { class: chrome_class(state, "command-rail"),
            Menubar {
                // The way to every simple command by name, in the slot the
                // catch-all ☰ menu held — the menu became a palette the day the
                // registry could list itself (`commands::ALL`).
                CommandSearch {}
                MenubarMenu { index: 0usize,
                    // What is on screen: the floating panels, and the chrome
                    // that stands outside their stack. Each entry wears its own
                    // mark — a panel's is the one its title bar wears — so the
                    // menu is a picture of the window rather than a list of its
                    // nouns (`PanelId::glyph`, `Command::icon`).
                    MenubarTrigger { {icon_large(icons::PANELS)} }
                    MenubarContent {
                        // One loop over one list (`commands::VisibilityToggle`),
                        // which is where what the menu holds — and in what order
                        // — is written down. Every row is a registry command, so
                        // the same act is reachable by search and by a chord of
                        // the user's own, and the row adds nothing the registry
                        // does not already carry.
                        for (i, entry) in VisibilityToggle::ALL.into_iter().enumerate() {
                            CmdItem { index: i, command: entry.command() }
                        }
                    }
                }
                // This client's preferences. A plain button inside the rail rather
                // than a third `MenubarMenu`: it opens a dialog on the click, so
                // there is no dropdown for the menubar to manage.
                button {
                    class: "rail-button",
                    // The menubar's own triggers carry `role="menuitem"`; matching
                    // it keeps the rail a well-formed menubar rather than a
                    // menubar with a stray button in it.
                    role: "menuitem",
                    r#type: "button",
                    title: Command::Settings.name(),
                    onclick: move |_| Command::Settings.run(state),
                    {icon_large(Command::Settings.icon())}
                }
            }
        }
        if show_new_doc() {
            NewDocumentModal { on_close: move |_| show_new_doc.set(false) }
        }
        if show_session() {
            collab::SessionModal { on_close: move |_| show_session.set(false) }
        }
        if show_export() {
            files::ExportModal { on_close: move |_| show_export.set(false) }
        }
        if show_settings() {
            SettingsModal { on_close: move |_| show_settings.set(false) }
        }
        if show_timing() {
            timings::TimingModal { on_close: move |_| show_timing.set(false) }
        }
        if show_credits() {
            CreditsModal { on_close: move |_| show_credits.set(false) }
        }
        if show_drag_presets() {
            drags::DragPresetModal { on_close: move |_| show_drag_presets.set(false) }
        }
    }
}

/// One row of the rail's menus, rendered from the command it runs
/// (`crate::commands`): the word, the mark, the shortcut column, the greyed
/// state and the mode tick (§18.2.4) are all the registry's, so what the menu
/// shows and what a click does cannot drift — and a chord a command gains is
/// advertised here without the row changing. The one prop left is the menu's
/// own business: `index` is the menubar's roving-focus order.
#[component]
fn CmdItem(index: usize, command: Command) -> Element {
    let state = use_context::<AppState>();
    // One memo per row, and rows are components, so each re-renders when its
    // own answer changes rather than on every commit (`state::use_obs`'s
    // argument, one field per component instead of a shared tuple).
    //
    // Both facts in the **one** memo, which is what that argument asks for: the
    // tick was read straight out of the registry in the body below, and
    // `Command::active` asks the projection for the three shape tools
    // (`commands::armed`) — so a row for one of those was subscribed to every
    // engine write however narrow its `enabled` memo was.
    let look = use_obs_opt(state, move |o| (command.enabled(o), command.active(state)));
    let (enabled, active) = look();
    rsx! {
        MenubarItem {
            index,
            value: format!("{command:?}"),
            disabled: !enabled,
            on_select: move |_| command.run(state),
            // The terse word, not the full name: a menu's trigger already
            // names the subject, which is `word`'s whole remit — the Panels
            // menu says "Color" where the palette must say "Color panel".
            span { class: "menu-item", {icon(command.icon())} {command.word()} }
            if let Some(chord) = command.shortcut(&state.bindings.read()) {
                span { class: "menu-shortcut", {chord} }
            }
            if let Some(on) = active {
                span { class: "menu-check",
                    if on { {icon(icons::CHECK)} }
                }
            }
        }
    }
}

/// The command search (§11): the rail's first entry, and the way to every
/// simple command by name. It opens like the menu beside it and stands in the
/// same spot, but the keyboard goes to a **field**, resting on the file family
/// (`commands::BASIC`) and narrowing to `commands::search` as the query grows.
/// Arrows move the highlight, Enter runs it, Escape puts the palette away; a
/// row is the same row the menus draw, printed from the same registry.
///
/// Our own dropdown rather than a third `MenubarMenu`, and not for styling: the
/// primitive's trigger light-dismisses its menu the moment DOM focus leaves it
/// for anything but a menu item, and the whole point of this surface is that
/// focus lives in a text field the primitive has never heard of. So it is
/// `panels::filter::AddFilterButton`'s arrangement instead — rows act on
/// `pointerdown`, dismissal is `onfocusout` — with one addition that pattern
/// never needed: focus moving *within* the palette (the trigger handing the
/// field the keyboard on open) must not read as leaving, so the handler asks
/// the event where focus went (`platform::focus_stays_within`).
#[component]
fn CommandSearch() -> Element {
    let state = use_context::<AppState>();
    let mut open = use_signal(|| false);
    let mut query = use_signal(String::new);
    // The highlighted row, moved by the arrows and spent by Enter. An index
    // into `shown`, reset with the query it indexes into.
    let mut sel = use_signal(|| 0usize);
    // The command whose shortcut is being recaptured, if any — armed by its
    // row's chip ([`BindChip`]), spent by the next chord the field hears.
    let mut capturing: Signal<Option<Command>> = use_signal(|| None);
    // The palette's own DOM node, held for exactly one question: did that
    // focusout land inside me.
    let mut root: Signal<Option<Event<MountedData>>> = use_signal(|| None);
    // The field's node, so a chip click can hand the keyboard back to it —
    // the chord about to be pressed must land where the capture listens.
    let mut field: Signal<Option<Event<MountedData>>> = use_signal(|| None);
    let shown = use_memo(move || commands::search(&query.read()));

    rsx! {
        div {
            class: "command-search",
            onmounted: move |e| root.set(Some(e)),
            onfocusout: move |e| {
                if !platform::focus_stays_within(root.read().as_ref(), &e) {
                    open.set(false);
                }
            },
            button {
                class: "rail-button",
                // `role` for the ⚙'s reason: the rail is a menubar, and this
                // keeps it one rather than a menubar with a stray button in it.
                role: "menuitem",
                r#type: "button",
                title: "Search commands",
                onclick: move |_| {
                    let show = !open();
                    // A fresh open is a fresh question: the resting offer, not
                    // whatever was typed — or half-captured — before the last
                    // dismissal.
                    if show {
                        query.set(String::new());
                        sel.set(0);
                        capturing.set(None);
                    }
                    open.set(show);
                },
                {icon_large(icons::SEARCH)}
            }
            if open() {
                div { class: "command-palette",
                    input {
                        class: "palette-field",
                        r#type: "text",
                        placeholder: "Search commands",
                        value: "{query}",
                        // The field takes the keyboard the moment it exists —
                        // the palette is *for* typing, and `input`'s window
                        // shortcuts already stand aside for a text field
                        // (`platform::KeyEvent::on_text_entry`).
                        onmounted: move |e| {
                            platform::focus(&e);
                            field.set(Some(e));
                        },
                        oninput: move |e| {
                            query.set(e.value());
                            sel.set(0);
                        },
                        onkeydown: move |e| {
                            // While a capture is armed, every keystroke is the
                            // capture's: none may reach the query, and none the
                            // browser (`commands::capture` says what one means).
                            if let Some(command) = capturing() {
                                e.prevent_default();
                                let m = e.modifiers();
                                let code = e.code().to_string();
                                match commands::capture(
                                    accel(m),
                                    m.contains(Modifiers::SHIFT),
                                    m.contains(Modifiers::ALT),
                                    &e.key(),
                                    &code,
                                ) {
                                    commands::Capture::Chord(chord) => {
                                        commands::rebind(state, command, chord);
                                        capturing.set(None);
                                    }
                                    commands::Capture::Clear => {
                                        commands::unbind(state, command);
                                        capturing.set(None);
                                    }
                                    commands::Capture::Cancel => capturing.set(None),
                                    commands::Capture::Pending => {}
                                }
                                return;
                            }
                            let count = shown.read().len();
                            match e.key() {
                                Key::Escape => open.set(false),
                                Key::Enter => {
                                    let pick = shown.read().get(sel()).copied();
                                    if let Some(command) = pick {
                                        run_from_palette(state, open, command);
                                    }
                                }
                                // The arrows move the highlight, not the caret:
                                // the field is one line, so the caret has no
                                // vertical to spend them on.
                                Key::ArrowDown => {
                                    if count > 0 {
                                        sel.set((sel() + 1).min(count - 1));
                                    }
                                    e.prevent_default();
                                }
                                Key::ArrowUp => {
                                    sel.set(sel().saturating_sub(1));
                                    e.prevent_default();
                                }
                                _ => {}
                            }
                        },
                    }
                    for (i, command) in shown.read().iter().copied().enumerate() {
                        PaletteRow {
                            key: "{command:?}",
                            command,
                            selected: i == sel(),
                            open,
                            capturing,
                            field,
                        }
                    }
                    if shown.read().is_empty() {
                        div { class: "palette-empty", "Nothing matches" }
                    }
                }
            }
        }
    }
}

/// One row of the palette, rendered from the command it runs — the mark, the
/// name, the greyed state, the live-act tint, and the chip that rebinds its
/// chord ([`BindChip`]).
///
/// **A component and not an inline `button`**, for [`CmdItem`]'s reason and one
/// more of its own. `Command::enabled` reads the projection and `Command::active`
/// reads it too for the shape tools (`commands::armed`), so written inline both
/// ran in the palette's own body — which subscribed the whole surface, field and
/// all, to every engine write, and re-ran `active` twice per row on each of them.
/// A row that owns its own memo re-renders when *its* answer moves and sleeps
/// through the rest, which is the arrangement the rail's menu has always had.
///
/// `selected` stays a prop rather than a memo: the highlight is the palette's
/// state, moved by the arrow keys, and the parent is the only thing that knows
/// where it is.
#[component]
fn PaletteRow(
    command: Command,
    selected: bool,
    open: Signal<bool>,
    capturing: Signal<Option<Command>>,
    field: Signal<Option<Event<MountedData>>>,
) -> Element {
    let state = use_context::<AppState>();
    let look = use_obs_opt(state, move |o| (command.enabled(o), command.active(state)));
    let (enabled, active) = look();
    rsx! {
        button {
            class: if selected { "palette-row selected" } else { "palette-row" },
            // Greyed by attribute, not by a native `disabled`, though the row
            // refuses to run either way (`run_from_palette`): the trailing chip
            // must stay clickable — a shortcut is rebindable whether or not the
            // document offers the act right now, and whether Undo has anything
            // to undo is no fact about its chord.
            "data-disabled": !enabled,
            // `pointerdown`, not `click`, for the filter picker's reason: it
            // beats the blur that would fold the palette away under the pointer.
            onpointerdown: move |_| run_from_palette(state, open, command),
            // A live act wears the select blue on its mark (`Command::active`) —
            // Share while a session runs — where a toggle's "you are in it" is
            // the tick the menu draws.
            span {
                class: "menu-item",
                class: if active == Some(true) { "cmd-active" },
                class: if active == Some(false) { "cmd-inactive" },
                {icon(command.icon())}
                {command.name()}
            }
            BindChip { command, capturing, field }
        }
    }
}

/// Run a palette row, palette closed first — a command may mount a dialog, and
/// the palette has no business outliving the choice. Refused whole while the
/// projection greys the row ([`Command::enabled`]): the row is not natively
/// disabled — its chip must stay live for rebinding — so this guard is the
/// entire refusal, for the pointer and for Enter alike, and a refused click
/// leaves the palette standing rather than closing on nothing.
fn run_from_palette(state: AppState, mut open: Signal<bool>, command: Command) {
    if !command.enabled(state.obs.peek().as_ref()) {
        return;
    }
    open.set(false);
    command.run(state);
}

/// A palette row's trailing shortcut, which is also the door to changing it:
/// the chord as a clickable chip, a hover-revealed `+` where there is none yet,
/// or the capture prompt while this row is the one listening. Click, then press
/// the new chord — the field keeps the keyboard and reads it
/// (`commands::capture`), so picking a binding is the same gesture as using one.
///
/// The one chip that only prints is Import's: its Ctrl+V is the browser's
/// paste, true whatever the table says, so offering to move it would be
/// offering a lie ([`Command::rebindable`]).
#[component]
fn BindChip(
    command: Command,
    capturing: Signal<Option<Command>>,
    field: Signal<Option<Event<MountedData>>>,
) -> Element {
    let state = use_context::<AppState>();
    let mut capturing = capturing;
    let grab = move |e: Event<PointerData>| {
        // The chip's press is the chip's alone: stopped so the row under it
        // does not run, default-prevented so focus never leaves the field —
        // which is where the chord about to be pressed must land.
        e.stop_propagation();
        e.prevent_default();
        capturing.set(Some(command));
        if let Some(f) = field.read().as_ref() {
            platform::focus(f);
        }
    };
    rsx! {
        if capturing() == Some(command) {
            span {
                class: "menu-shortcut bind-chip capturing",
                title: "Press the new shortcut \u{2014} Backspace removes it, \
                        Escape keeps what was there",
                "press keys\u{2026}"
            }
        } else if !command.rebindable() {
            if let Some(chord) = command.shortcut(&state.bindings.read()) {
                span { class: "menu-shortcut", title: "The browser's own paste", {chord} }
            }
        } else if let Some(chord) = command.shortcut(&state.bindings.read()) {
            span {
                class: "menu-shortcut bind-chip",
                title: "Click, then press the new shortcut",
                onpointerdown: grab,
                {chord}
            }
        } else {
            span {
                class: "menu-shortcut bind-chip bind-add",
                title: "Add a shortcut: click, then press it",
                onpointerdown: grab,
                {icon(icons::ADD)}
            }
        }
    }
}
